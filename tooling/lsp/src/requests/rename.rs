use std::{
    collections::HashMap,
    future::{self, Future},
};

use async_lsp::{ErrorCode, ResponseError};
use fm::FileMap;
use lsp_types::{
    PrepareRenameResponse, RenameParams, TextDocumentPositionParams, TextEdit, Url, WorkspaceEdit,
};
use nargo::insert_all_files_for_workspace_into_file_manager;
use noirc_driver::file_manager_with_stdlib;
use noirc_errors::Location;
use noirc_frontend::macros_api::NodeInterner;

use crate::{parse_diff, resolve_workspace_for_source_path, LspState};

use super::{position_to_byte_index, to_lsp_location};

pub(crate) fn on_prepare_rename_request(
    state: &mut LspState,
    params: TextDocumentPositionParams,
) -> impl Future<Output = Result<Option<PrepareRenameResponse>, ResponseError>> {
    let result = process_rename_request(state, params, |search_for_location, interner, _| {
        let rename_possible = interner.check_rename_possible(search_for_location);
        Some(PrepareRenameResponse::DefaultBehavior { default_behavior: rename_possible })
    });
    future::ready(result)
}

pub(crate) fn on_rename_request(
    state: &mut LspState,
    params: RenameParams,
) -> impl Future<Output = Result<Option<WorkspaceEdit>, ResponseError>> {
    let result = process_rename_request(
        state,
        params.text_document_position,
        |search_for_location, interner, files| {
            let rename_changes =
                interner.find_rename_symbols_at(search_for_location).map(|locations| {
                    let rs = locations.iter().fold(
                        HashMap::new(),
                        |mut acc: HashMap<Url, Vec<TextEdit>>, location| {
                            let file_id = location.file;
                            let span = location.span;

                            let Some(lsp_location) = to_lsp_location(files, file_id, span) else {
                                return acc;
                            };

                            let edit = TextEdit {
                                range: lsp_location.range,
                                new_text: params.new_name.clone(),
                            };

                            acc.entry(lsp_location.uri).or_default().push(edit);

                            acc
                        },
                    );
                    rs
                });

            let response = WorkspaceEdit {
                changes: rename_changes,
                document_changes: None,
                change_annotations: None,
            };

            Some(response)
        },
    );
    future::ready(result)
}

fn process_rename_request<F, T>(
    state: &mut LspState,
    text_document_position_params: TextDocumentPositionParams,
    callback: F,
) -> Result<T, ResponseError>
where
    F: FnOnce(Location, &NodeInterner, &FileMap) -> T,
{
    let file_path =
        text_document_position_params.text_document.uri.to_file_path().map_err(|_| {
            ResponseError::new(ErrorCode::REQUEST_FAILED, "URI is not a valid file path")
        })?;

    let workspace = resolve_workspace_for_source_path(file_path.as_path()).unwrap();
    let package = workspace.members.first().unwrap();

    let package_root_path: String = package.root_dir.as_os_str().to_string_lossy().into();

    let mut workspace_file_manager = file_manager_with_stdlib(&workspace.root_dir);
    insert_all_files_for_workspace_into_file_manager(&workspace, &mut workspace_file_manager);
    let parsed_files = parse_diff(&workspace_file_manager, state);

    let (mut context, crate_id) =
        crate::prepare_package(&workspace_file_manager, &parsed_files, package);

    let interner;
    if let Some(def_interner) = state.cached_definitions.get(&package_root_path) {
        interner = def_interner;
    } else {
        // We ignore the warnings and errors produced by compilation while resolving the definition
        let _ = noirc_driver::check_crate(&mut context, crate_id, false, false, false);
        interner = &context.def_interner;
    }

    let files = context.file_manager.as_file_map();
    let file_id = context.file_manager.name_to_id(file_path.clone()).ok_or(ResponseError::new(
        ErrorCode::REQUEST_FAILED,
        format!("Could not find file in file manager. File path: {:?}", file_path),
    ))?;
    let byte_index =
        position_to_byte_index(files, file_id, &text_document_position_params.position).map_err(
            |err| {
                ResponseError::new(
                    ErrorCode::REQUEST_FAILED,
                    format!("Could not convert position to byte index. Error: {:?}", err),
                )
            },
        )?;

    let search_for_location = noirc_errors::Location {
        file: file_id,
        span: noirc_errors::Span::single_char(byte_index as u32),
    };

    Ok(callback(search_for_location, interner, files))
}

#[cfg(test)]
mod rename_tests {
    use super::*;
    use crate::test_utils;
    use lsp_types::{Position, Range, WorkDoneProgressParams};
    use tokio::test;

    async fn check_rename_succeeds(directory: &str, name: &str) {
        let (mut state, noir_text_document) = test_utils::init_lsp_server(directory).await;

        let main_path = noir_text_document.path();

        // First we find out all of the occurrences of `name` in the main.nr file.
        // Note that this only works if that name doesn't show up in other places where we don't
        // expect a rename, but we craft our tests to avoid that.
        let file_contents = std::fs::read_to_string(main_path)
            .unwrap_or_else(|_| panic!("Couldn't read file {}", main_path));
        let file_lines: Vec<&str> = file_contents.lines().collect();
        let ranges: Vec<_> = file_lines
            .iter()
            .enumerate()
            .filter_map(|(line_num, line)| {
                line.find(name).map(|index| {
                    let start = Position { line: line_num as u32, character: index as u32 };
                    let end =
                        Position { line: line_num as u32, character: (index + name.len()) as u32 };
                    Range { start, end }
                })
            })
            .collect();

        // Test renaming works on any instance of the symbol.
        for target_range in &ranges {
            let target_position = target_range.start;

            let params = RenameParams {
                text_document_position: TextDocumentPositionParams {
                    text_document: lsp_types::TextDocumentIdentifier {
                        uri: noir_text_document.clone(),
                    },
                    position: target_position,
                },
                new_name: "renamed_function".to_string(),
                work_done_progress_params: WorkDoneProgressParams { work_done_token: None },
            };

            let response = on_rename_request(&mut state, params)
                .await
                .expect("Could not execute on_prepare_rename_request")
                .unwrap();

            let changes = response.changes.expect("Expected to find rename changes");
            let mut changes: Vec<Range> =
                changes.values().flatten().map(|edit| edit.range).collect();
            changes.sort_by_key(|range| range.start.line);
            assert_eq!(changes, ranges);
        }
    }

    #[test]
    async fn test_on_prepare_rename_request_cannot_be_applied() {
        let (mut state, noir_text_document) = test_utils::init_lsp_server("rename_function").await;

        let params = TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier { uri: noir_text_document },
            position: lsp_types::Position { line: 0, character: 0 }, // This is at the "f" of an "fn" keyword
        };

        let response = on_prepare_rename_request(&mut state, params)
            .await
            .expect("Could not execute on_prepare_rename_request");

        assert_eq!(
            response,
            Some(PrepareRenameResponse::DefaultBehavior { default_behavior: false })
        );
    }

    #[test]
    async fn test_rename_function() {
        check_rename_succeeds("rename_function", "another_function").await;
    }

    #[test]
    async fn test_rename_qualified_function() {
        check_rename_succeeds("rename_qualified_function", "bar").await;
    }

    #[test]
    async fn test_rename_function_in_use_statement() {
        check_rename_succeeds("rename_function_use", "some_function").await;
    }

    #[test]
    async fn test_rename_struct() {
        check_rename_succeeds("rename_struct", "Foo").await;
    }
}
